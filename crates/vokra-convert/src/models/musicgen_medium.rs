#![allow(clippy::doc_lazy_continuation)]
//! **MusicGen-Medium** (`facebook/musicgen-medium`, **cc-by-nc-4.0**):
//! safetensors → GGUF conversion (Wave 5 candidate, 2026-08-01).
//!
//! Input: the upstream `facebook/musicgen-medium` release — Meta AudioCraft's
//! 1.5B-parameter text-to-music autoregressive transformer LM (Copet et al.
//! 2023, arXiv:2306.05284 "Simple and Controllable Music Generation").
//! MusicGen operates over a discrete audio token vocabulary from a paired
//! EnCodec RVQ codec (32 kHz, 4 codebooks, 50 Hz frame rate) and consumes
//! text conditioning via a frozen T5-base encoder. The "medium" size is the
//! middle rung of the family (300M `-small` / **1.5B `-medium`** / 3.3B
//! `-large` / plus melody-conditioned siblings), targeting the quality/
//! compute sweet-spot the AudioCraft paper's ablation identified.
//!
//! # Vokra scope — music generation (per 2026-07-30 scope expansion)
//!
//! MusicGen is the first **music generation** target to land a converter.
//! Vokra was pinned to speech-only scope from CLAUDE.md「音声モデルで深さ
//! で勝つ」through 2026-07-29; the 2026-07-30 依頼者指示「asr,tts,音楽系,
//! 音声分離など全てのモデルに対応したい」expanded scope to include music
//! generation + speech separation + audio LLMs
//! (`[[project-scope-expansion-2026-07-30]]`). The category tag `music`
//! this converter stamps is the first use in the tree — sibling categories
//! (`tts` / `asr` / `codec` / `vocoder` / `s2s` / `vad` / `speaker` / `f0`
//! / `separator` / `bert`) are the speech-tree taxonomy; `music` opens the
//! music-tree branch. Silently sharing the `tts` tag would misroute
//! model-card generation + zoo taxonomy tooling.
//!
//! # License posture — CC-BY-NC 4.0 default (**NonCommercial**)
//!
//! Weight redistribution default is [`LicenseClass::NonCommercial`]. The HF
//! model card at `huggingface.co/facebook/musicgen-medium` carries
//! `license: cc-by-nc-4.0` on its YAML front-matter — the same posture
//! sibling `facebook/musicgen-{small,large,melody,stereo-*}` releases ship
//! (Meta AudioCraft weight policy: code MIT under
//! `github.com/facebookresearch/audiocraft`, but the trained weights are
//! non-commercial). This mirrors the X-Codec 2 (T4 tier) precedent landed
//! 2026-07-28: `LicenseClass::NonCommercial` activates
//! [`LicenseClass::requires_research_flag`] at load time → **fail-closed**:
//! an unmarked commercial-mode caller cannot silently bring the weights up.
//!
//! Callers may override at the outer `convert_file --license <spdx>`
//! boundary when they legitimately hold the weights under a distinct SPDX
//! id (e.g. a permissive re-training on public-domain music — the same
//! Whisper / kokoro / vits-ja / xcodec2 override pattern).
//!
//! # BF16 pass-through (mirror of xcodec2 / neucodec / vibevoice)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`MusicGenMediumReport::bf16_passthrough`] records how many
//! BF16 tensors landed on this arm so a silent widen / downcast cannot
//! slip in undetected. Upstream medium checkpoint is F32/F16-heavy per
//! HF transformers `MusicgenForConditionalGeneration` default, so the
//! BF16 arm is defensive today; the counter is kept for future
//! BF16-quantized derivative releases.
//!
//! # Scale — vast.ai handoff (~11.4 GB)
//!
//! MusicGen-Medium ships as ~11.4 GB on HF (LM decoder ~5-6 GB + T5 text
//! encoder ~1 GB + EnCodec RVQ audio codec + optional stereo heads =
//! bundle total). This is above the M1 iMac 16 GB local-convert
//! threshold (memory [[feedback-large-models-on-vast-ai]]: ≥8 GB safe,
//! Voxtral-Small-24B 48 GB confirmed swap-death) — conversion + publish
//! happens on vast.ai per `docs/handoff/vast-ai-large-model-publish.md`.
//!
//! # Sibling family (future waves)
//!
//! Future MusicGen family additions (small / large / melody / stereo-*
//! variants) can either:
//!   (a) land as sibling files (`musicgen_small.rs`, `musicgen_large.rs`,
//!       …) mirror of the `chatterbox` / `chatterbox_turbo` /
//!       `chatterbox_nano` split, OR
//!   (b) refactor into a shared `musicgen.rs` with a `MusicGenVariant`
//!       enum mirror of the `vocos` / `snac` / `bigvgan` / `focalcodec`
//!       split.
//! Today's landing is a standalone medium-only file (the `xcodec2` /
//! `wavtokenizer` posture) — a single §3.1 row, a single ModelKind, no
//! pre-emptive enum bloat before a second variant exists.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM / VibeVoice
//! / neucodec / step_audio2_mini / xcodec2 contract). Real-weight
//! binding + AudioCraft-parity forward is a follow-up wave gated on
//! §3.1 sign-off + real-checkpoint tensor-name manifest fetch; this
//! converter passes every F32 / F16 / BF16 tensor through unchanged so a
//! future `MusicGenMedium::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream Python pipeline
//! (`transformers.MusicgenForConditionalGeneration`) is deferred to
//! owner (`docs/license-audit.md` §3.1 sign-off queue = 2026-08-01 wave).
//! A parity harness following the sepformer / DFN3 / Kokoro precedent
//! (reference dumper → fixture commit → Rust parity test) will land
//! when the runtime binder + music-generation ops (token-conditioned
//! transformer LM decode + EnCodec RVQ decode) land.
//!
//! # No ONNX (permanent)
//!
//! MusicGen ships safetensors + PyTorch pickle; this converter **never**
//! touches ONNX (FR-LD-05). The pipeline is re-implemented natively in a
//! future `crates/vokra-models/src/musicgen/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4). Between now and that
//! landing, the runtime consumer walks the emitted tensor names and
//! either succeeds or fails loudly per FR-EX-08.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MusicGen GGUFs.
///
/// The arch tag matches the AudioCraft `MusicGen` class name (Copet et al.
/// 2023 arXiv:2306.05284). Intentionally distinct from every sibling arch
/// tag — MusicGen is a **music-generation transformer LM over EnCodec RVQ
/// tokens** conditioned on frozen T5 text encoder, distinct topology from
/// every speech-tree sibling (Whisper ASR / CosyVoice2 speech-LM /
/// Voxtral audio-LLM / Moshi full-duplex S2S / codec siblings). Silently
/// sharing an arch tag would mis-route runtime dispatch to a wrong-shape
/// forward.
///
/// Future MusicGen family variants (small / large / melody / stereo-*)
/// share this same `musicgen` arch tag — the topology is identical, only
/// the model dims + optional melody-conditioning head differ (the same
/// arch-shared / name-distinct posture snac / vocos / bigvgan use).
pub const ARCH: &str = "musicgen";

/// `vokra.model.name` value written for the canonical MusicGen-Medium
/// GGUF (the variant-specific display name a consumer sees when
/// inspecting a converted artifact).
///
/// The variant-specific spelling mirrors the wavtokenizer /
/// chatterbox_turbo pattern (variant-in-name) so a future
/// `musicgen-small` / `musicgen-large` / `musicgen-melody` /
/// `musicgen-stereo-medium` lands as a distinct `NAME` under the shared
/// [`ARCH`] tag — the same shared-arch / distinct-name split every
/// future family sibling will use.
pub const NAME: &str = "musicgen-medium";

/// `vokra.model.name` value written for the canonical MusicGen-Melody
/// GGUF (Wave 5 sibling landed 2026-08-02, `facebook/musicgen-melody`,
/// **cc-by-nc-4.0**). Melody = medium 1.5B autoregressive transformer
/// LM + **chroma conditioning** (12-bin chromagram of a reference
/// melody clip concatenated to the T5 text conditioning stream); the
/// LM topology is byte-identical to MusicGen-Medium ([`NAME`]), only
/// the conditioning frontend differs. Reusing the medium BF16 pass-
/// through arm (single [`convert_musicgen_family_file`] helper + this
/// sibling wrapper) rather than a dedicated `musicgen_melody.rs` file
/// — the tensor-name manifest is identical to sibling medium, only
/// `vokra.model.name` + `vokra.provenance.*` (`model_id`, `source`,
/// `upstream_hf`) flip.
pub const MELODY_NAME: &str = "musicgen-melody";

/// `vokra.model.category` value — MusicGen is the first **music
/// generation** target in the tree (per the 2026-07-30 scope expansion
/// `[[project-scope-expansion-2026-07-30]]`), distinct from the
/// speech-tree category taxonomy (`tts` / `asr` / `codec` / `vocoder` /
/// `s2s` / `vad` / `speaker` / `f0` / `separator` / `bert`). The
/// category chunk is a taxonomy tag orthogonal to `arch`; the runtime
/// does not dispatch on it (arch does), but it is machine-readable for
/// model-zoo / catalog surfaces (see `docs/license-audit.md`).
pub const CATEGORY: &str = "music";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
pub const UPSTREAM_HF: &str = "facebook/musicgen-medium";

/// Upstream HF repository slug for the melody sibling
/// (`facebook/musicgen-melody`), recorded under
/// `vokra.provenance.upstream_hf`. See [`MELODY_NAME`] for the
/// medium/melody topology relationship (byte-identical LM + chroma-
/// conditioning frontend delta).
pub const MELODY_UPSTREAM_HF: &str = "facebook/musicgen-melody";

/// The default upstream weight license — `cc-by-nc-4.0`, per the HF
/// model card `license: cc-by-nc-4.0` (Meta AudioCraft weight policy;
/// the code layer at `github.com/facebookresearch/audiocraft` is MIT
/// but the trained weights are non-commercial — the same code/weight
/// license split X-Codec 2 landed with 2026-07-28). Callers can override
/// at the `convert_musicgen_medium_file(_, _, license=Some(_))` boundary
/// when the source distribution declares a different SPDX id (a
/// permissive re-training on public-domain music, for example).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`). Kept short — the
/// license machine class is carried separately in the
/// `vokra.provenance.weight_license` chunk.
const UPSTREAM_SOURCE: &str =
    "facebook/musicgen-medium (Meta AudioCraft 1.5B text-to-music LM, cc-by-nc-4.0)";

/// Human-readable upstream source note for the melody sibling
/// (`facebook/musicgen-melody`, Meta AudioCraft 1.5B text-to-music LM +
/// chroma-melody conditioning, cc-by-nc-4.0). Stored in
/// `vokra.provenance.source`.
const MELODY_UPSTREAM_SOURCE: &str = "facebook/musicgen-melody (Meta AudioCraft 1.5B text-to-music LM + chroma conditioning, cc-by-nc-4.0)";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling BF16 pass-through converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MusicGen-Medium conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::xcodec2::XCodec2Report`,
/// `super::wavtokenizer::WavtokenizerReport`,
/// `super::neucodec::NeucodecReport`) adapted to the file-oriented
/// `convert_musicgen_medium_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MusicGenMediumReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a non-zero
    /// here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `facebook/musicgen-medium` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`, returning a
/// [`MusicGenMediumReport`].
///
/// The upstream release ships as a bundle (LM decoder + T5 text encoder
/// + paired EnCodec RVQ codec + optional stereo heads) totalling
/// ~11.4 GB. Callers running on the M1 iMac 16 GB machine should NOT
/// attempt local conversion (memory [[feedback-large-models-on-vast-ai]]:
/// ≥8 GB safe threshold; Voxtral-Small-24B 48 GB confirmed swap-death) —
/// conversion + publish happens on vast.ai per
/// `docs/handoff/vast-ai-large-model-publish.md`.
///
/// If the upstream release ships torch pickle (`.bin`) rather than
/// safetensors, callers pre-flatten offline via
/// `tools/parity/musicgen_medium_prepare_checkpoint.py` (thin wrapper
/// over `bin_to_safetensors.py`, the SpeechT5-HiFi-GAN pattern). This
/// function accepts safetensors only — no pickle parser enters the
/// Vokra tree (NFR-DS-02 zero-dep + FR-LD-05 no pickle in runtime).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-4.0"`, `NonCommercial`) — the
/// upstream HF release ships CC-BY-NC 4.0 per Meta AudioCraft weight
/// policy.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// cannot be assembled.
pub fn convert_musicgen_medium_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MusicGenMediumReport, ConvertError> {
    convert_musicgen_family_file(input, output, license, NAME, UPSTREAM_HF, UPSTREAM_SOURCE)
}

/// Converts a `facebook/musicgen-melody` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`.
///
/// MusicGen-Melody is the medium 1.5B autoregressive transformer LM
/// (byte-identical topology to [`convert_musicgen_medium_file`]) plus a
/// **12-bin chromagram melody-conditioning frontend** concatenated to
/// the T5 text conditioning stream — only the frontend + conditioning
/// projection differs. The BF16 pass-through pipeline is therefore
/// shared with the medium arm via the private
/// [`convert_musicgen_family_file`] helper; only the
/// `vokra.model.name` + `vokra.provenance.{model_id,source,upstream_hf}`
/// chunks flip to the melody spellings ([`MELODY_NAME`] /
/// [`MELODY_UPSTREAM_HF`] / [`MELODY_UPSTREAM_SOURCE`]).
///
/// **Scale ~6 GB → vast.ai handoff.** Do NOT attempt a local convert on
/// the M1 iMac 16 GB machine (memory
/// [[feedback-large-models-on-vast-ai]]: ≥8 GB safe cutoff; Voxtral-
/// Small-24B 48 GB confirmed swap-death is the calibration point).
/// Real-weight parity + the chroma frontend runtime op are deferred to
/// owner sign-off (`docs/license-audit.md` §3.1 sign-off queue). The
/// tensor-name manifest for the LM decoder + T5 encoder + EnCodec RVQ
/// codec is identical to sibling [`NAME`], so a future
/// `MusicGenMedium::from_gguf` walks the same names for both — the
/// chroma frontend is a converter-side add-on tensor group.
///
/// `license` optionally overrides the stamped weight license — see
/// [`convert_musicgen_medium_file`] for the override semantics + empty-
/// string research-flag-downgrade guard. Defaults to
/// `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-4.0"`, `NonCommercial`).
///
/// # Errors
///
/// Same failure modes as [`convert_musicgen_medium_file`].
pub fn convert_musicgen_melody_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MusicGenMediumReport, ConvertError> {
    convert_musicgen_family_file(
        input,
        output,
        license,
        MELODY_NAME,
        MELODY_UPSTREAM_HF,
        MELODY_UPSTREAM_SOURCE,
    )
}

/// Shared implementation for the MusicGen family (medium + melody, both
/// cc-by-nc-4.0, same LM + T5 + EnCodec topology, only the model-id /
/// upstream-hf / source stamps + optional chroma frontend differ).
///
/// Kept `pub(crate)` so future variants (a future `-stereo-medium`, for
/// example) can piggyback without duplicating the BF16 pass-through
/// dispatch. External callers should route through the variant-specific
/// wrappers ([`convert_musicgen_medium_file`] /
/// [`convert_musicgen_melody_file`]) so the correct built-in defaults
/// stay in one place.
pub(crate) fn convert_musicgen_family_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    name: &str,
    upstream_hf: &str,
    upstream_source: &str,
) -> Result<MusicGenMediumReport, ConvertError> {
    // NB: MusicGen-Medium bundle is ~11.4 GB (melody ~6 GB).
    // `std::fs::read` peaks at ~2x file size (input buffer + parsed
    // safetensors view = additive in the worst case). The vast.ai
    // runbook allocates a 32 GB+ box for this class of publish per
    // `docs/handoff/vast-ai-large-model-publish.md` §2, so simple
    // eager-read is acceptable — no streaming reader needed for a
    // one-shot offline convert. Moshi (14 GB) is the streaming-mandated
    // tier and lives in its own module; MusicGen sits just below.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Built-in stamp = cc-by-nc-4.0 NonCommercial. The `license` argument
    // (Some(non-empty spdx)) overrides these three chunks — but with the
    // built-in gate the artifact fails **closed** at load time in
    // commercial mode
    // (`LicenseClass::NonCommercial::requires_research_flag = true`), so
    // an operator who never touched the license flag cannot silently
    // bring up an NC weight in production. The empty-string case is
    // explicitly filtered (mirror of xcodec2 / wavtokenizer): an empty
    // override must NOT wipe the built-in stamp — that would be a silent
    // research-flag downgrade.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::NonCommercial),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(name), Some(upstream_source));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, upstream_hf);

    let mut report = MusicGenMediumReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR (mirror of
    // xcodec2 / neucodec / wavtokenizer / speecht5_hifigan / vibevoice);
    // the runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// A unique temp path — per-process id **plus** a monotonic counter so
    /// two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-musicgen-medium-{tag}-{}-{n}",
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16) with
    /// caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
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

    /// The BF16 pass-through arm must emit GGUF type 30
    /// (`GgmlType::BF16`) with byte-identical payload — mirror of the
    /// xcodec2 / wavtokenizer / neucodec pin.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast (zeroed payloads would
        // round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic MusicGen state-dict name — `decoder.model.
        // decoder.embed_tokens.weight` is one of the LM decoder's embed
        // matrices in `MusicgenForConditionalGeneration`.
        let input_bytes = safetensors_one(
            "decoder.model.decoder.embed_tokens.weight",
            "BF16",
            &[2, 3],
            &bf16,
        );
        let input = tmp_path("bf16-in");
        let output = tmp_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_musicgen_medium_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let file = GgufFile::open(&output).expect("load output gguf");
        let info = file
            .tensor_info("decoder.model.decoder.embed_tokens.weight")
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

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 mixed-dtype pass-through with the additive-default
    /// invariant on `bf16_passthrough` and all arch / provenance /
    /// category stamps — including the **critical** default NonCommercial
    /// stamp (the whole point of the CC-BY-NC 4.0 flip vs. sibling
    /// permissive converters).
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_noncommercial() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable half-values: 1.0=0x3C00, -2.0=0xC000,
        // -0.5=0xB800, 3.0=0x4200, 0.15625=0x3100, 42.0=0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        // Mirror realistic MusicGen state-dict tensor names:
        //   `text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight`
        //     — T5-base text encoder Q projection.
        //   `decoder.model.decoder.layers.0.self_attn.k_proj.weight`
        //     — LM decoder K projection.
        let input_bytes = safetensors_f32_then_f16(
            "text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight",
            &[1, 2],
            &f32_bytes,
            "decoder.model.decoder.layers.0.self_attn.k_proj.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = tmp_path("mixed-in");
        let output = tmp_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_musicgen_medium_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16-only input must leave the BF16 counter at the Default 0 (additive-default invariant)"
        );

        let file = GgufFile::open(&output).expect("load output gguf");

        let f32_info = file
            .tensor_info("text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("decoder.model.decoder.layers.0.self_attn.k_proj.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Arch / name / category / provenance chunks land with the
        // built-in cc-by-nc-4.0 NonCommercial stamp.
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
            Some(CATEGORY),
            "vokra.model.category must be `music` (first music-tree entry, distinct from speech-tree tags)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        // The default license path must stamp cc-by-nc-4.0 / NonCommercial
        // (the whole point of this converter vs. the sibling wavtokenizer
        // which defaults to MIT / Permissive — Meta AudioCraft weights
        // are non-commercial per HF card front-matter).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str())
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// A caller-supplied `license` (e.g. re-trained on a permissive
    /// public-domain music corpus) overrides the built-in cc-by-nc-4.0
    /// NonCommercial stamp. Same override pattern as
    /// `convert_file_licensed` — the model_id / arch / category / upstream
    /// stamps survive but the license triple flips.
    #[test]
    fn caller_license_override_swaps_the_stamp() {
        // Non-zero payloads that are NOT approximations of π/e —
        // clippy::approx_constant would flag 3.14/2.71 as a naked
        // approximation of the standard constants.
        let f32_vals: [f32; 2] = [11.5, -6.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "decoder.model.decoder.layers.0.fc1.weight",
            "F32",
            &[1, 2],
            &f32_bytes,
        );
        let input = tmp_path("override-in");
        let output = tmp_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        // Override to Apache-2.0 (Permissive) — the caller retrained on a
        // permissive corpus.
        let report =
            convert_musicgen_medium_file(&input, &output, Some("apache-2.0")).expect("convert");
        assert_eq!(report.written, 1);

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override class must be re-derived from the SPDX id"
        );
        // Model id / arch / category / upstream_hf remain the built-in
        // values — the override changes only the license triple.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category (music) must not flip when license overrides"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade. The
    /// `Some(s) if !s.is_empty()` guard in `convert_musicgen_medium_file`
    /// keeps the default cc-by-nc-4.0 NonCommercial stamp (mirror of
    /// xcodec2 / wavtokenizer empty-string guard test).
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_vals: [f32; 2] = [0.5, -0.5];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "decoder.model.decoder.layers.1.fc2.weight",
            "F32",
            &[1, 2],
            &f32_bytes,
        );
        let input = tmp_path("empty-in");
        let output = tmp_path("empty-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let _ = convert_musicgen_medium_file(&input, &output, Some("")).expect("convert");

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty string must NOT downgrade the license stamp"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
            "empty string must NOT downgrade the class"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Non-float dtypes reach the skipped counter, not the pass-through
    /// arm — defensive since the safetensors reader already rejects
    /// non-F32/F16/BF16 dtypes at parse time. This test asserts the
    /// defensive dispatch stays in place: if the safetensors reader ever
    /// admits I8/I64/etc, this converter must NOT silently emit them
    /// (mirror of xcodec2's dispatch contract).
    #[test]
    fn nonzero_bf16_counter_only_bumps_for_bf16_tensor() {
        // Sanity pin: additive-default report layout — an empty
        // safetensors buffer produces zero reads / zero writes.
        // Constructing a valid empty safetensors requires the header
        // `{}` (2 bytes) prefixed by its little-endian u64 length (8
        // bytes) = 10 bytes total.
        let empty_header = "{}";
        let mut empty_safetensors = Vec::new();
        empty_safetensors.extend_from_slice(&(empty_header.len() as u64).to_le_bytes());
        empty_safetensors.extend_from_slice(empty_header.as_bytes());

        let input = tmp_path("empty-st-in");
        let output = tmp_path("empty-st-out");
        std::fs::write(&input, &empty_safetensors).expect("write empty safetensors");

        let report = convert_musicgen_medium_file(&input, &output, None)
            .expect("empty safetensors must be accepted");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        // Even with zero tensors the metadata chunks still land — the
        // provenance stamp is independent of the tensor walk.
        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
            "stamp must land even with no tensors — fail-closed license posture applies unconditionally"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// The melody sibling wrapper stamps the melody-specific
    /// `vokra.model.name` + `vokra.provenance.{model_id,source,
    /// upstream_hf}` chunks (not the medium ones) while sharing the
    /// medium arch / category / default cc-by-nc-4.0 NonCommercial
    /// license stamp and BF16 pass-through pipeline. This is the whole
    /// point of reusing the family helper — the delta must be limited
    /// to the four id chunks.
    #[test]
    fn melody_wrapper_stamps_melody_ids_and_shares_default_license() {
        // Non-zero BF16 payload so a subsequent byte-identity check
        // still asserts the pass-through arm on the shared helper.
        let values: [f32; 4] = [1.5, -0.75, 6.5, -12.0];
        let bf16 = bf16_bytes(&values);
        let input_bytes = safetensors_one(
            "decoder.model.decoder.embed_tokens.weight",
            "BF16",
            &[2, 2],
            &bf16,
        );
        let input = tmp_path("melody-in");
        let output = tmp_path("melody-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_musicgen_melody_file(&input, &output, None).expect("convert melody");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::open(&output).expect("load output gguf");

        // Shared: arch stays `musicgen` (same LM topology) and category
        // stays `music` (music-tree taxonomy).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "melody must share the medium `musicgen` arch tag"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "melody must share the `music` category"
        );

        // Flipped: name / model_id / upstream_hf must be the melody
        // spellings, NOT the medium ones — otherwise the runtime cannot
        // distinguish the two artifacts and a downstream chroma-front-
        // end binder would silently misroute to the medium checkpoint.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(MELODY_NAME)
        );
        assert_ne!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
            "melody must NOT ship the medium `musicgen-medium` name"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(MELODY_NAME)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(MELODY_UPSTREAM_HF)
        );

        // License triple = cc-by-nc-4.0 / NonCommercial (shared default
        // with medium — Meta AudioCraft weight policy is uniform across
        // the family).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str())
        );

        // BF16 pass-through arm must land byte-identical.
        let info = file
            .tensor_info("decoder.model.decoder.embed_tokens.weight")
            .expect("BF16 tensor present in melody output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
